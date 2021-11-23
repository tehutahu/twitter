import requests


class LineNotify:
    def __init__(self, token: str, notify_api: str) -> None:
        self.line_notify_token = token
        self.line_notify_api = notify_api
        self.headers = {
          "Authorization": f"Bearer {self.line_notify_token}"
        }

    def send(self, msg, image=None):
        data = { "message": f" {msg}" }
        if image:
            files = {"imageFile": image}
        else:
            files = None
        requests.post(self.line_notify_api, headers = self.headers, data=data, files=files)