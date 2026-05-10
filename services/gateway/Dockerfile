FROM nginx:1.27-alpine
RUN rm -f /etc/nginx/conf.d/default.conf
COPY services/gateway/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
