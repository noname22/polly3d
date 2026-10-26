//! `polly` — Polly3D MCP server over stdio.
//!
//! Usage: polly [--viewer] [--project FILE]

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use polly_core::document::Document;
use polly_mcp::PollyServer;
use polly_mcp::services::Services;
use rmcp::ServiceExt;

struct Args {
    viewer: bool,
    project: Option<std::path::PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { viewer: false, project: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--viewer" => a.viewer = true,
            "--project" => a.project = Some(it.next().context("--project needs a file")?.into()),
            "-h" | "--help" => {
                eprintln!("polly [--viewer] [--project FILE]\n\nPolly3D low-poly modeller, an MCP server on stdio.");
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
    let mut doc = Document::new();
    let services = Arc::new(Services::default());
    if let Some(p) = &args.project {
        if p.exists() {
            let bytes = std::fs::read(p)?;
            doc.replace(polly_export::load_project(&bytes)?);
        }
        services.settings.lock().unwrap().project_path = Some(p.clone());
    }
    let doc = Arc::new(Mutex::new(doc));
    let server = PollyServer::new(doc.clone(), services.clone());

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let serve = async move {
        let running = server.serve(rmcp::transport::stdio()).await?;
        running.waiting().await?;
        anyhow::Ok(())
    };
    if args.viewer {
        // The window event loop must own the main thread; MCP runs on the runtime.
        let handle = rt.spawn(serve);
        let _ = handle;
        tracing::warn!("viewer not available yet");
        rt.block_on(async { futures_pending().await });
        Ok(())
    } else {
        rt.block_on(serve)
    }
}

async fn futures_pending() {
    std::future::pending::<()>().await
}
