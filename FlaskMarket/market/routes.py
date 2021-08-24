from market import app
from flask import render_template, request
from market.models import Item

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')
@app.route('/market')
def market_page():
    items = Item.query.all()
    return render_template('market.html',items=items)
@app.route('/perfil', methods=['GET','POST'])
def perfil_page():
    if request.method == 'POST':
        print(request.form.getlist('mycheckbox'))
        return 'Done'
    return render_template('perfil.html')


