#!/bin/bash
set -e

SETTINGS_FILE="/etc/searxng/settings.yml"
SECRET_PLACEHOLDER="CHANGE_ME_TO_A_LONG_RANDOM_VALUE"

if [ -f "$SETTINGS_FILE" ]; then
    if grep -q "$SECRET_PLACEHOLDER" "$SETTINGS_FILE"; then
        # Generate a random secret key
        NEW_SECRET=$(openssl rand -hex 32)
        # Replace the placeholder with the new secret
        sed -i "s/$SECRET_PLACEHOLDER/$NEW_SECRET/g" "$SETTINGS_FILE"
        echo "Generated new secret key for SearXNG"
    fi
fi

exec "$@"
