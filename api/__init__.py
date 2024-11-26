from flask import Flask

def create_app():
    from api.server import create_app
    return create_app() 