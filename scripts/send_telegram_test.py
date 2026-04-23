from __future__ import annotations

import argparse

from runtime.telegram_alerts import get_telegram_config, send_telegram_test_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Telegram test message.")
    parser.add_argument(
        "--message",
        default="Test alert from Trading Assistant",
        help="Message text to send.",
    )
    args = parser.parse_args()

    cfg = get_telegram_config()
    if not cfg.enabled:
        print("Telegram alerts are disabled. Enable TELEGRAM_ALERTS_ENABLED or runtime config first.")
        return
    if not cfg.configured:
        print("Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    ok = send_telegram_test_message(message=args.message)
    print("sent" if ok else "failed")


if __name__ == "__main__":
    main()
