import asyncio
import logging
import sqlite3
import asqlite
import twitchio
from twitchio.ext import commands
from twitchio import eventsub
from .tui import GladosUI

LOGGER: logging.Logger = logging.getLogger("TwitchBot")

# Note: this uses twitchio 3.0, please follow the startup tutorial here:
# https://twitchio.dev/en/latest/getting-started/quickstart.html
# You should have the file "tokens.db" in this project's root folder
CLIENT_ID: str = (
    ""  # The CLIENT ID from the Twitch Dev Console
)
CLIENT_SECRET: str = (
    ""  # The CLIENT SECRET from the Twitch Dev Console
)
# Note: use https://www.streamweasels.com/tools/convert-twitch-username-%20to-user-id/ to convert UserName to Twitch ID
BOT_ID = ""  # The Account ID of the bot user...
OWNER_ID = ""  # Your personal User ID..
STREAM_KEY = ""


class TwitchBot(commands.Bot):
    def __init__(
        self,
        glados_ui: GladosUI,
        token_database: asqlite.Pool,
        prefix: str,
    ):
        self.glados_ui = glados_ui
        self.token_database = token_database
        super().__init__(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=BOT_ID,
            owner_id=OWNER_ID,
            prefix=prefix,
        )

    async def setup_hook(self) -> None:
        # Add our component which contains our commands...
        await self.add_component(TwitchChatListener(self))

        # Subscribe to read chat (event_message) from our channel as the bot...
        # This creates and opens a websocket to Twitch EventSub...
        subscription = eventsub.ChatMessageSubscription(
            broadcaster_user_id=OWNER_ID, user_id=BOT_ID
        )
        await self.subscribe_websocket(payload=subscription)

    async def add_token(
        self, token: str, refresh: str
    ) -> twitchio.authentication.ValidateTokenPayload:
        # Make sure to call super() as it will add the tokens internally and return us some data...
        resp: twitchio.authentication.ValidateTokenPayload = await super().add_token(
            token, refresh
        )

        # Store our tokens in a simple SQLite Database when they are authorized...
        query = """
        INSERT INTO tokens (user_id, token, refresh)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET
            token = excluded.token,
            refresh = excluded.refresh;
        """

        async with self.token_database.acquire() as connection:
            await connection.execute(query, (resp.user_id, token, refresh))

        LOGGER.info("Added token to the database for user: %s", resp.user_id)
        return resp

    async def load_tokens(self, path: str | None = None) -> None:
        # We don't need to call this manually, it is called in .login() from .start() internally...

        async with self.token_database.acquire() as connection:
            rows: list[sqlite3.Row] = await connection.fetchall(
                """SELECT * from tokens"""
            )

        for row in rows:
            await self.add_token(row["token"], row["refresh"])

    async def setup_database(self) -> None:
        # Create our token table, if it doesn't exist..
        query = """CREATE TABLE IF NOT EXISTS tokens(user_id TEXT PRIMARY KEY, token TEXT NOT NULL, refresh TEXT NOT NULL)"""
        async with self.token_database.acquire() as connection:
            await connection.execute(query)

    async def event_ready(self) -> None:
        LOGGER.info("Successfully logged in as: %s", self.bot_id)


class TwitchChatListener(commands.Component):
    def __init__(self, bot: TwitchBot):
        self.bot = bot

    # @commands.Component.listener()
    # async def event_message(self, payload: twitchio.ChatMessage) -> None:
    #     self.bot.glados_ui.send_message_to_llm(
    #         user_input=payload.text, user_name=payload.chatter.name
    #     )

    @commands.command(aliases=["chat"])
    async def hi(self, ctx: commands.Context) -> None:
        """Command to interact with GlaDOS!

        !chat
        """
        message: str = ctx.message.text.strip()
        if message.startswith("!chat"):
            message = message[len("!chat") :].strip()
        self.bot.glados_ui.send_message_to_llm(
            user_input=message, user_name=ctx.chatter.display_name
        )


async def start_twitch_bot(glados_ui: GladosUI):
    twitchio.utils.setup_logging(level=logging.INFO)

    async def runner() -> None:
        async with asqlite.create_pool("tokens.db") as tdb:
            bot = TwitchBot(
                glados_ui=glados_ui,
                token_database=tdb,
                prefix="!",
            )
            await bot.setup_database()  # Ensure the database is set up
            await bot.start()

    try:
        await runner()
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down due to KeyboardInterrupt...")


if __name__ == "__main__":
    asyncio.run(start_twitch_bot(GladosUI()))
