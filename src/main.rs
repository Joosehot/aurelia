//! Aurelia: a block of ring accelerators that answers from branded knowledge.
//!
//!   aurelia chat [--viz] [--seed N]    talk in the terminal (/why shows the last answer's rings)
//!   aurelia ask "question" ... [--seed N]   answer lines one after another, as one conversation
//!   aurelia debug [RUN_DIR]            which rings peaked and what they held, for the last answer
//!   aurelia replay RUN_DIR N           replay answer N from its snapshot and seed
//!   aurelia serve [--port 5830]        the chat in the browser, at http://127.0.0.1:5830

mod block;
mod engine;
mod ethics;
mod infer;
mod knowledge;
mod store;
mod think;
mod wordnet;

use anyhow::{bail, Context, Result};
use block::{Block, Cfg};
use engine::{Engine, Reply, Snapshot};
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Default)]
struct Config {
    wordnet: Option<String>,
    #[serde(default)]
    block: Cfg,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn config() -> Result<Config> {
    let path = root().join("aurelia.toml");
    if !path.exists() {
        return Ok(Config::default());
    }
    Ok(toml::from_str(&std::fs::read_to_string(&path)?).context("reading aurelia.toml")?)
}

fn load(cfg: &Config) -> Result<wordnet::WordNet> {
    let dir = cfg.wordnet.clone().unwrap_or(r"C:\Users\joose\Primogenitus\motor\data\wordnet\dict".into());
    let t = std::time::Instant::now();
    let wn = wordnet::WordNet::read(Path::new(&dir)).with_context(|| format!("reading the word network at {dir}"))?;
    eprintln!("word network: {} senses, {} words ({:.1} s)", wn.synsets.len(), wn.index.len(), t.elapsed().as_secs_f32());
    Ok(wn)
}

fn load_store() -> Result<store::Store> {
    let t = std::time::Instant::now();
    let s = store::Store::load(&root().join("knowledge"))?;
    eprintln!("world and book knowledge: {} items ({:.1} s)", s.items.len(), t.elapsed().as_secs_f32());
    Ok(s)
}

fn new_run_dir() -> PathBuf {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    root().join("runs").join("chat").join(format!("{secs}"))
}

fn latest_run_dir() -> Result<PathBuf> {
    let base = root().join("runs").join("chat");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&base)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs.pop().context("no chat yet")
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("chat");
    match cmd {
        "chat" => chat(args.contains(&"--viz".to_string()), arg_value(&args, "--seed").map(|s| s.parse()).transpose()?),
        "ask" => {
            let seed: Option<u64> = arg_value(&args, "--seed").map(|s| s.parse()).transpose()?;
            let mut lines = Vec::new();
            let mut skip = false;
            for a in &args[1..] {
                if skip {
                    skip = false;
                } else if a == "--seed" {
                    skip = true;
                } else {
                    lines.push(a.clone());
                }
            }
            ask(&lines, seed)
        }
        "serve" => serve(arg_value(&args, "--port").map(|p| p.parse()).transpose()?.unwrap_or(5830)),
        "debug" => {
            let dir = match args.get(1) {
                Some(d) => PathBuf::from(d),
                None => latest_run_dir()?,
            };
            let r: Reply = serde_json::from_slice(&std::fs::read(dir.join("last.json"))?)?;
            print_debug(&r);
            Ok(())
        }
        "replay" => {
            let (Some(dir), Some(n)) = (args.get(1), args.get(2)) else { bail!("aurelia replay RUN_DIR N") };
            replay(Path::new(dir), n.parse()?)
        }
        _ => bail!("commands: chat [--viz] [--seed N] | ask \"line\" ... | debug [RUN_DIR] | replay RUN_DIR N"),
    }
}

fn engine() -> Result<Engine> {
    let cfg = config()?;
    let wn = load(&cfg)?;
    let dir = new_run_dir();
    Engine::new(wn, load_store()?, cfg.block, dir, root().join("brand"))
}

fn chat(viz: bool, seed: Option<u64>) -> Result<()> {
    let mut e = engine()?;
    println!("Aurelia: {} rings, {} particles. Ask me about a word, tell me about yourself. /why shows the last answer's rings, /quit ends.", e.block.rings(), e.block.parts.len());
    println!("log: {}", e.dir.display());
    let stdin = std::io::stdin();
    let mut seed = seed;
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line == "/why" || line == "/debug" {
            match &e.last {
                Some(r) => print_debug(r),
                None => println!("no answer yet"),
            }
            continue;
        }
        let s = match seed.take() {
            Some(s) => s,
            None => engine::fresh_seed(),
        };
        let r = if viz {
            let mut draw = |b: &Block, t: u32, total: u32| frame(b, t, total);
            e.answer(line, s, Some(&mut draw))?
        } else {
            e.answer(line, s, None)?
        };
        println!("Aurelia: {}", r.text);
        println!("         (reply {} · seed {})", r.n, r.seed);
    }
    Ok(())
}

fn ask(lines: &[String], seed: Option<u64>) -> Result<()> {
    let mut e = engine()?;
    println!("log: {}", e.dir.display());
    for (i, l) in lines.iter().enumerate() {
        let s = seed.map(|s| s + i as u64).unwrap_or_else(engine::fresh_seed);
        let t = std::time::Instant::now();
        let r = e.answer(l, s, None)?;
        println!("> {l}");
        println!("Aurelia: {}", r.text);
        println!(
            "         (reply {} · seed {} · {} steps · {} collisions · {} charged rings · {:.0} ms)",
            r.n,
            r.seed,
            r.steps,
            r.collisions,
            r.charged_rings,
            t.elapsed().as_secs_f32() * 1000.0
        );
    }
    Ok(())
}

fn debug_text(r: &Reply) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "reply {} · seed {} · \"{}\"", r.n, r.seed, r.message);
    let _ = writeln!(
        s,
        "wave from ring {:?}, {} steps, {} collisions; {} rings charged, {} rings peaked in all",
        r.origin, r.steps, r.collisions, r.charged_rings, r.peaked_rings
    );
    if r.read.is_empty() {
        let _ = writeln!(s, "no charged ring was read");
    }
    for p in &r.read {
        let _ = writeln!(s, "ring {:?}: peaked at step {}, limit {:.2}", p.ring, p.peak_step, p.limit);
        for (id, c, strike, text, ok) in &p.facts {
            if *id == u32::MAX {
                let _ = writeln!(s, "   {strike}: {text}");
                continue;
            }
            let _ = writeln!(s, "   node {id} charge {c:.2} {} {}", if *ok { "checked" } else { "FAILED CHECK, not said" }, text);
            let _ = writeln!(s, "      from {strike}");
        }
    }
    let _ = writeln!(s, "reply: {}", r.text);
    s
}

fn print_debug(r: &Reply) {
    print!("{}", debug_text(r));
}

/// A chat page in the browser: the branded block stays loaded, one answer at a time.
fn serve(port: u16) -> Result<()> {
    use std::io::Read;
    let mut e = engine()?;
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).with_context(|| format!("port {port} is taken"))?;
    println!("Aurelia chat: http://127.0.0.1:{port}  (log: {})", e.dir.display());
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        // the head, then as much body as it says
        let (head, body) = loop {
            let n = match s.read(&mut chunk) {
                Ok(0) | Err(_) => break (String::new(), Vec::new()),
                Ok(n) => n,
            };
            buf.extend_from_slice(&chunk[..n]);
            if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..at]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                    .unwrap_or(0);
                let mut body = buf[at + 4..].to_vec();
                while body.len() < len {
                    match s.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => body.extend_from_slice(&chunk[..n]),
                    }
                }
                break (head, body);
            }
        };
        let first = head.lines().next().unwrap_or("").to_string();
        let (status, kind, out): (&str, &str, Vec<u8>) = if first.starts_with("GET / ") || first.starts_with("GET /index") {
            ("200 OK", "text/html; charset=utf-8", include_str!("chat.html").as_bytes().to_vec())
        } else if first.starts_with("POST /ask") {
            let text = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(|t| t.to_string()))
                .unwrap_or_default();
            let t0 = std::time::Instant::now();
            match e.answer(&text, engine::fresh_seed(), None) {
                Ok(r) => {
                    let j = serde_json::json!({ "text": r.text, "n": r.n, "seed": r.seed.to_string(),
                        "ms": t0.elapsed().as_millis() as u64, "why": debug_text(&r) });
                    println!("> {text}\nAurelia: {}", r.text);
                    ("200 OK", "application/json", j.to_string().into_bytes())
                }
                Err(err) => ("500 Internal Server Error", "application/json", serde_json::json!({ "text": format!("error: {err}"), "n": 0, "seed": "", "ms": 0, "why": "" }).to_string().into_bytes()),
            }
        } else {
            ("404 Not Found", "text/plain", b"not here".to_vec())
        };
        let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", out.len());
        let _ = s.write_all(&out);
    }
    Ok(())
}

fn replay(dir: &Path, n: u32) -> Result<()> {
    let text = std::fs::read_to_string(dir.join("replies.jsonl"))?;
    let orig: Reply = text
        .lines()
        .filter_map(|l| serde_json::from_str::<Reply>(l).ok())
        .find(|r| r.n == n)
        .with_context(|| format!("no reply {n} in {}", dir.display()))?;
    let snap: Snapshot = serde_json::from_slice(&std::fs::read(dir.join(&orig.snapshot))?)?;
    let cfg = config()?;
    let wn = load(&cfg)?;
    let mut e = Engine::new(wn, load_store()?, cfg.block, dir.join("replay"), root().join("brand"))?;
    e.persist = false;
    e.load_snapshot(snap);
    let r = e.answer(&orig.message, orig.seed, None)?;
    println!("> {}", orig.message);
    println!("then:   {}", orig.text);
    println!("replay: {}", r.text);
    println!("{}", if r.text == orig.text { "the same answer" } else { "DIFFERENT" });
    Ok(())
}

/// A top view of the block: each column shows its highest ring's rise toward
/// its limit; @ marks a charged ring that has reached its limit.
fn frame(b: &Block, t: u32, total: u32) {
    if t % 4 != 0 && t != total {
        return;
    }
    let ramp: Vec<char> = " .:-=+*#".chars().collect();
    let mut out = String::from("\x1b[H");
    for j in (0..b.ry).step_by(2) {
        for i in 0..b.rx {
            let mut best = 0.0f32;
            let mut lit = false;
            let mut charged = false;
            for z in 0..b.rz {
                let r = b.ring(i, j, z);
                best = best.max(b.fill(r));
                charged |= b.limit[r] > 1.0;
                lit |= b.limit[r] > 1.0 && b.peak[r] >= 0;
            }
            if lit {
                out.push_str("\x1b[93m@\x1b[0m");
            } else if charged {
                out.push_str("\x1b[96mo\x1b[0m");
            } else {
                out.push(ramp[((best * (ramp.len() - 1) as f32) as usize).min(ramp.len() - 1)]);
            }
        }
        out.push('\n');
    }
    out.push_str(&format!("step {t}/{total} · collisions {}   \n", b.collisions));
    if t <= 4 {
        print!("\x1b[2J");
    }
    print!("{out}");
    let _ = std::io::stdout().flush();
    std::thread::sleep(std::time::Duration::from_millis(15));
}
