//! `polly` — Polly3D MCP server over stdio.
//!
//! Usage: polly [--viewer | --no-viewer] [--project FILE]

use std::sync::Arc;

use anyhow::{Context, Result};
use polly_mcp::PollyServer;
use polly_mcp::services::Services;
use polly_mcp::workspace::{SceneSlot, Workspace};
use rmcp::ServiceExt;

mod anim_panel;
mod viewer;

struct Args {
    /// Open the viewer window at startup (otherwise only on `show_viewer`).
    viewer: bool,
    /// Disable the viewer entirely (no display connection).
    no_viewer: bool,
    project: Option<std::path::PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { viewer: false, no_viewer: false, project: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--viewer" => a.viewer = true,
            "--no-viewer" => a.no_viewer = true,
            "--project" => a.project = Some(it.next().context("--project needs a file")?.into()),
            "-h" | "--help" => {
                eprintln!(
                    "polly [--viewer | --no-viewer] [--project FILE]\n\n\
                     Polly3D low-poly modeller, an MCP server on stdio.\n\
                     The live viewer window opens when the agent calls show_viewer;\n\
                     --viewer opens it at startup, --no-viewer disables it."
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument '{other}'"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    // stdout is the MCP channel: log to stderr only.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "polly=info,warn".into()))
        .init();
    let args = parse_args()?;
    let services = Arc::new(Services::default());
    let mut main_scene = SceneSlot::default();
    if let Some(p) = &args.project {
        if p.exists() {
            let bytes = std::fs::read(p)?;
            main_scene.doc.replace(polly_export::load_project(&bytes)?);
        }
        main_scene.project_path = Some(p.clone());
    }
    let workspace = Arc::new(Workspace::with_main(main_scene));
    // The window event loop has to live on the main thread for the whole
    // process; it costs nothing while no window is open.
    let event_loop = if args.no_viewer {
        None
    } else {
        match viewer::event_loop() {
            Ok(el) => {
                let proxy = el.create_proxy();
                services.set_viewer(Box::new(move |cmd| {
                    proxy.send_event(cmd).map_err(|_| anyhow::anyhow!("the viewer has shut down"))
                }));
                Some(el)
            }
            Err(e) => {
                tracing::info!("{e:#}; show_viewer will be unavailable");
                None
            }
        }
    };
    let server = PollyServer::new(workspace.clone(), services.clone());

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let serve = async move {
        let running = server.serve(rmcp::transport::stdio()).await?;
        running.waiting().await?;
        anyhow::Ok(())
    };
    match event_loop {
        Some(el) => {
            // MCP runs on the runtime; when the client disconnects the whole
            // process exits, window included.
            rt.spawn(async move {
                if let Err(e) = serve.await {
                    tracing::error!("mcp server: {e:#}");
                }
                std::process::exit(0);
            });
            if let Err(e) = viewer::run(el, workspace, args.viewer) {
                tracing::error!("viewer: {e:#}; continuing without it");
            }
            rt.block_on(std::future::pending::<()>());
            Ok(())
        }
        None => rt.block_on(serve),
    }
}
