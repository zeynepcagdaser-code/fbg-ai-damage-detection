from pyngrok import ngrok

token = '3DRGr5j6UWsrNtg8kZi8gO5Lpy2_T4v2dfYjSzC4oCDnsu2E'
ngrok.set_auth_token(token)
print('ngrok auth token ayarlandı')
