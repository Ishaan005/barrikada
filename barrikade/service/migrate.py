"""Run Barrikade-owned metadata migrations."""

from barrikade.service.config import ServiceSettings
from barrikade.service.storage import MetadataStore


def main() -> int:
    settings = ServiceSettings.from_env()
    MetadataStore(settings.database_url).migrate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
