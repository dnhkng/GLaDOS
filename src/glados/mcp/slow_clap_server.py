from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slow_clap_demo")


@mcp.tool()
def slow_clap(claps: int = 1) -> str:
    """Return a slow clap response for demo purposes."""
    try:
        claps = int(claps)
    except (TypeError, ValueError):
        claps = 1
    claps = max(1, min(claps, 5))
    return " ".join(["clap"] * claps)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
