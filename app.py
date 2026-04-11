from web.app import app, ensure_dirs


if __name__ == "__main__":
    # Root compatibility wrapper: delegate to the web package app.
    ensure_dirs()
    app.run(debug=True)