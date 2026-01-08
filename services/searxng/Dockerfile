FROM searxng/searxng:latest
COPY settings.yml /etc/searxng/settings.yml
COPY init_secret.sh /init_secret.sh
RUN chmod +x /init_secret.sh
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
ENTRYPOINT ["/init_secret.sh"]
CMD ["/usr/local/bin/docker-entrypoint.sh"]