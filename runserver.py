from waitress import serve

from BI_Chat_Analysis.wsgi import application


if __name__ == '__main__':
    serve(application, host = 'localhost', port='8000')
