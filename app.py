import json
import os

from flask import Flask
from flask import render_template
from flask import request
from flask import flash
from gensim.models import Word2Vec


app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))


def scale_simirarity(most_similar, max_size, min_size):
    words, similarities = zip(*most_similar)
    magnifi_max = max_size / similarities[0]
    magnifi_min = min_size / similarities[-1]
    magnifi_diff = (magnifi_max - magnifi_min) / (len(words) - 1)
    scaled = [similarity * ((len(words) - i - 1) * magnifi_diff + magnifi_min)
              for i, similarity in enumerate(similarities)]
    return list(zip(words, scaled))


@app.route('/', methods=['GET', 'POST'])
def index():
    if not 'model' in app.jinja_env.globals:
        with app.open_resource('jawiki-20160601_vec_mc20_dim200_pn.bin') as f:
            app.jinja_env.globals['model'] = Word2Vec.load_word2vec_format(
                f, binary=True, unicode_errors='replace')
        app.jinja_env.globals['words'] = list(
            app.jinja_env.globals['model'].vocab.keys())
        return render_template('layout.html')

    if request.method == 'GET':
        return render_template('layout.html')
    else:
        proper_noun = request.form['word']
        if proper_noun in app.jinja_env.globals['model'].vocab:
            most_similar = app.jinja_env.globals[
                'model'].most_similar([proper_noun], topn=40)
            most_similar = [[w, s] for w, s in most_similar]
            return render_template('layout.html', most_similar=most_similar, word=proper_noun, data=json.dumps(scale_simirarity(most_similar, 60, 10), ensure_ascii=False))
        else:
            flash('Out of vocabulary.')
            return render_template('layout.html', word=proper_noun)


if __name__ == '__main__':
    app.run()
