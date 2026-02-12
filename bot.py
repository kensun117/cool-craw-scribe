import logging
import os

import discord
from discord.ext import commands


def build_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.guilds = True
    intents.messages = True

    bot = commands.Bot(intents=intents)

    @bot.event
    async def on_ready() -> None:
        logging.getLogger(__name__).info("Bot online: %s (%s)", bot.user, bot.user.id if bot.user else "n/a")

    bot.load_extension("cogs.meeting_recorder")
    return bot


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Missing DISCORD_BOT_TOKEN environment variable.")

    bot = build_bot()
    bot.run(token)


if __name__ == "__main__":
    main()
