import cProfile
import pstats
import os

from eval.local_eval import run_game, default_agent_factory, REAL_GAMES_DIR

def main():
    game_file = os.path.join("eval", "real_games", "tr87", "cd924810", "tr87.py")
    if not os.path.exists(game_file):
        print(f"Game not found! {game_file}")
        return

    import threading, sys, traceback, time
    def dump():
        time.sleep(15)
        print("DUMPING MAIN THREAD STACK:", file=sys.stderr)
        frame = sys._current_frames()[threading.main_thread().ident]
        traceback.print_stack(frame, file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    threading.Thread(target=dump, daemon=True).start()

    # Run game with profiler
    print("Profiling...")
    cProfile.runctx(
        "run_game(game_file, default_agent_factory, max_actions=200, verbose=True)",
        globals(), locals(), "profile.stats"
    )

    # Print stats
    p = pstats.Stats("profile.stats")
    p.sort_stats('tottime').print_stats(20)

if __name__ == "__main__":
    main()
