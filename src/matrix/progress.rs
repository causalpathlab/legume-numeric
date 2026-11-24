//! Workspace-wide progress-bar style and the single [`MULTI_PROGRESS`] that
//! every bar registers with.
//!
//! This lives in `legume_numeric::matrix` — the lowest common dependency of `data-beans`,
//! `data_beans::alg`, `data_beans::aux`, and the binaries — so the whole
//! workspace shares ONE style definition and ONE `MultiProgress`. That single
//! `MultiProgress` is what lets `indicatif_log_bridge` (installed by
//! `data_beans::aux::logging::init_logger`) interleave `log` output cleanly
//! above the bars. Duplicating the bar/`MultiProgress` in another crate (as
//! `graph-embedding-util` once did) silently spawns a second, *unbridged*
//! `MultiProgress` whose bars corrupt the log output — so every crate must
//! draw through this module. `data_beans::aux::logging` re-exports these names
//! for backward compatibility.
//!
//! Off a terminal (an MCP server's child, a batch job's log) indicatif draws
//! nothing, so each bar made here is also said in a line on stderr every
//! [`EVERY`] while it moves, and once as it ends: see [`PREFIX`] for the
//! line's shape, which `legume-mcp` reads back into a job's progress.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle, WeakProgressBar};
use std::io::IsTerminal;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

/// The single global `MultiProgress` every workspace progress bar registers
/// with, so `log` output routed through `indicatif_log_bridge` interleaves
/// cleanly above the bars instead of corrupting them.
pub static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Standard bounded-bar template: `[elapsed] bar pos/len (eta) msg`. The
/// trailing `{msg}` is empty unless a caller sets one (e.g.
/// `new_progress_bar(n).with_message("blocks")`).
const BAR_TEMPLATE: &str = "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}";

/// Tick frames shared by every [`new_spinner`].
const SPINNER_TICKS: &str = "⠁⠂⠄⡀⢀⠠⠐⠈ ";

/// Repaint cadence for both bars and spinners. Without a steady tick a bar
/// only repaints when [`ProgressBar::inc`] fires, so work with a heavy-tailed
/// per-item cost (one huge gene among thousands of small ones) leaves a frozen
/// elapsed time on screen for minutes and reads as a hang.
const STEADY_TICK: Duration = Duration::from_millis(200);

/// Create a progress bar registered with the shared [`MULTI_PROGRESS`] and
/// styled with the standard template. Attach a trailing label with
/// [`ProgressBar::with_message`], e.g. `new_progress_bar(n).with_message("blocks")`.
/// Repaints on the shared `STEADY_TICK` cadence.
#[must_use]
pub fn new_progress_bar(len: u64) -> ProgressBar {
    let prog_bar = MULTI_PROGRESS.add(ProgressBar::new(len));
    prog_bar.set_style(
        ProgressStyle::with_template(BAR_TEMPLATE)
            .unwrap()
            .progress_chars("##-"),
    );
    prog_bar.enable_steady_tick(STEADY_TICK);
    lines::watch(&prog_bar);
    prog_bar
}

/// Create a spinner registered with the shared [`MULTI_PROGRESS`] for
/// unbounded / streaming work (no known total). `template` is an indicatif
/// spinner template (e.g. `"{spinner} streamed {pos} fragments ({per_sec})"`);
/// the shared tick frames and the shared `STEADY_TICK` cadence are applied so
/// the spinner animates and stays visually consistent across crates.
#[must_use]
pub fn new_spinner(template: &str) -> ProgressBar {
    let prog_bar = MULTI_PROGRESS.add(ProgressBar::new_spinner());
    prog_bar.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .tick_chars(SPINNER_TICKS),
    );
    prog_bar.enable_steady_tick(STEADY_TICK);
    lines::watch(&prog_bar);
    prog_bar
}

/// What starts each progress line said off a terminal. The rest of the line
/// is one JSON object:
///
/// ```text
/// legume-progress {"id":3,"label":"blocks","pos":12,"len":40,"elapsed_s":31,"eta_s":72,"done":false}
/// ```
///
/// `id` numbers the bars of one process (several may run at once), `label`
/// is the bar's message (may be empty), `len` and `eta_s` are `null` for a
/// spinner, and `done` is true on a bar's last line.
pub const PREFIX: &str = "legume-progress ";

/// How often a bar off a terminal says where it stands, while it moves;
/// `LEGUME_PROGRESS_EVERY` (seconds; 0 for never) sets it otherwise.
pub const EVERY: Duration = Duration::from_secs(15);

/// A bar as its line says it.
#[derive(Clone, Debug, PartialEq)]
struct Said {
    id: u64,
    label: String,
    pos: u64,
    len: Option<u64>,
    elapsed_s: u64,
    eta_s: Option<u64>,
    done: bool,
}

impl Said {
    fn of(id: u64, bar: &ProgressBar, done: bool) -> Self {
        let len = bar.length();
        Self {
            id,
            label: bar.message(),
            pos: bar.position(),
            len,
            elapsed_s: bar.elapsed().as_secs(),
            eta_s: len.map(|_| bar.eta().as_secs()),
            done,
        }
    }

    /// The line, [`PREFIX`] first.
    fn line(&self) -> String {
        let opt = |v: Option<u64>| v.map_or("null".to_string(), |v| v.to_string());
        format!(
            "{PREFIX}{{\"id\":{},\"label\":{},\"pos\":{},\"len\":{},\"elapsed_s\":{},\"eta_s\":{},\"done\":{}}}",
            self.id,
            json_string(&self.label),
            self.pos,
            opt(self.len),
            self.elapsed_s,
            opt(self.eta_s),
            self.done
        )
    }
}

/// `s` as a JSON string.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The bars said in lines off a terminal, and the thread that says them.
mod lines {
    use super::*;

    /// A bar watched: its number, a handle that lets it go, and what its
    /// last line said (none yet).
    struct Watched {
        id: u64,
        bar: WeakProgressBar,
        said: Option<Said>,
    }

    static WATCHED: Mutex<Vec<Watched>> = Mutex::new(Vec::new());
    static NEXT: AtomicU64 = AtomicU64::new(1);

    /// How often to say the bars, none when stderr is a terminal (the bars
    /// are drawn) or lines are turned off.
    fn every() -> Option<Duration> {
        static EVERY_SET: LazyLock<Option<Duration>> = LazyLock::new(|| {
            let every = match std::env::var("LEGUME_PROGRESS_EVERY") {
                Ok(s) => s.trim().parse::<f64>().ok().map(Duration::from_secs_f64),
                Err(_) => Some(EVERY),
            };
            every.filter(|d| !d.is_zero() && !std::io::stderr().is_terminal())
        });
        *EVERY_SET
    }

    /// Say `bar` in lines from now on, when they are said at all.
    pub(super) fn watch(bar: &ProgressBar) {
        let Some(every) = every() else { return };
        static THREAD: std::sync::Once = std::sync::Once::new();
        THREAD.call_once(|| {
            let spawned = std::thread::Builder::new()
                .name("progress-lines".into())
                .spawn(move || loop {
                    std::thread::sleep(every);
                    say();
                });
            if let Err(e) = spawned {
                log::warn!("progress lines: {e}");
            }
        });
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut watched = WATCHED.lock().unwrap_or_else(|e| e.into_inner());
        watched.push(Watched {
            id,
            bar: bar.downgrade(),
            said: None,
        });
    }

    /// The lines due now: each bar that moved since its last (or, every
    /// fourth time, that did not: it is still there), and a last line for
    /// each that ended; those that ended are no longer watched.
    pub(super) fn say() {
        static ROUND: AtomicU64 = AtomicU64::new(0);
        let round = ROUND.fetch_add(1, Ordering::Relaxed);
        let mut watched = WATCHED.lock().unwrap_or_else(|e| e.into_inner());
        watched.retain_mut(|w| {
            let (now, keep) = match w.bar.upgrade() {
                Some(bar) if bar.is_finished() => (Said::of(w.id, &bar, true), false),
                Some(bar) => (Said::of(w.id, &bar, false), true),
                // Let go unfinished (a parallel iterator's bar, dropped as it
                // ends): last said as it last stood.
                None => match w.said.take() {
                    Some(last) => (Said { done: true, ..last }, false),
                    None => return false,
                },
            };
            let moved = w.said.as_ref().is_none_or(|s| {
                (s.pos, s.len, &s.label, s.done) != (now.pos, now.len, &now.label, now.done)
            });
            if moved || !keep || round % 4 == 3 {
                eprintln!("{}", now.line());
            }
            w.said = Some(now);
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bar's line is the prefix and one JSON object, its label escaped.
    #[test]
    fn a_line_is_the_prefix_and_json() {
        let said = Said {
            id: 3,
            label: "a \"b\"\\c".into(),
            pos: 12,
            len: Some(40),
            elapsed_s: 31,
            eta_s: Some(72),
            done: false,
        };
        assert_eq!(
            said.line(),
            r#"legume-progress {"id":3,"label":"a \"b\"\\c","pos":12,"len":40,"elapsed_s":31,"eta_s":72,"done":false}"#
        );
        let spinner = Said {
            len: None,
            eta_s: None,
            done: true,
            label: String::new(),
            ..said
        };
        assert!(spinner
            .line()
            .ends_with(r#""len":null,"elapsed_s":31,"eta_s":null,"done":true}"#));
    }
}
