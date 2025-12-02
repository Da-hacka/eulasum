import flask
from AI_model import Model
import bs4
import requests
import os


class Gui:
    def __init__(self):
        self.model = None
        self.app = flask.Flask(__name__)
        self._setup_routes()

    def get_model(self):
        if self.model is None:
            self.model = Model()
        return self.model

    def _setup_routes(self):
        @self.app.route("/", methods=["GET", "POST"])
        def home():
            summary = ""
            risk = None

            if flask.request.method == "POST":
                raw_input = flask.request.form.get("eula_input", "").strip()
                try:
                    if self.is_url(raw_input):
                        eula_text = self.fetch_clean_text(raw_input)
                    else:
                        eula_text = raw_input

                    summary, risk = self.get_model().grab_sum(eula_text)

                except Exception as e:
                    summary = f"Error during summarization:\n{e}"
                    risk = None

            return self.render_page(summary, risk)

    def run(self):
        port = int(os.environ.get("PORT", 10000))
        self.app.run(host="0.0.0.0", port=port)

    # ---------------------------------------------------------
    # ✅ HTML PAGE WITH SAFETY INDICATOR
    # ---------------------------------------------------------
    def render_page(self, summary_text, risk_score):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        .smooth-button {
          background-color: #393E46;
          color: white;
          border: none;
          padding: 10px 20px;
          font-size: 16px;
          border-radius: 5px;
          cursor: pointer;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
          transition: all 0.3s ease;
        }
        .smooth-button:hover {
          background-color: #45a049;
          transform: scale(1.05);
        }
        .indicator {
          width: 150px;
          height: 30px;
          border-radius: 5px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: bold;
          margin-top: 10px;
        }
        </style>
        </head>

        <body style="background-color:#222831; color:#DFD0B8; font-family: 'Comfortaa', sans-serif;">
            <h1>EULASum: Summarize with Ease</h1>

            <form method="post">
                <label>Enter EULA text or URL:</label><br>
                <textarea name="eula_input" rows="10" cols="80"
                    style="background-color:#393E46; color:#DFD0B8;"></textarea><br><br>
                <button class="smooth-button" type="submit">Generate Summary</button>
            </form>

            {% if risk_score is not none %}
                <h2>Safety Indicator:</h2>
                <div class="indicator"
                     style="background-color: {{ 'green' if risk_score < 40 else 'red' }};">
                    {{ 'SAFE' if risk_score < 40 else 'RISKY' }}
                </div>
            {% endif %}

            <h2>Summary Output:</h2>
            <textarea readonly rows="15" cols="80"
                style="background-color:#393E46; color:#DFD0B8;">{{ summary_text }}</textarea>
        </body>
        </html>
        """

        return flask.render_template_string(html, summary_text=summary_text, risk_score=risk_score)

    def is_url(self, text):
        return text.strip().lower().startswith("http")

    def fetch_clean_text(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")