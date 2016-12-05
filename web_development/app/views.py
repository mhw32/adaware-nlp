from flask import render_template
from app import app


@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Mike Man'}

    return render_template("index.html",
                           )
