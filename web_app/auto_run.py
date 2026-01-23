import webbrowser
import subprocess
import time
import sys
import os
from threading import Timer

def open_browser():
    """Open the web browser after a delay"""
    # Wait for the server to start
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:5001')

def main():
    # Open browser in a separate thread after a delay
    timer = Timer(1.0, open_browser)
    timer.start()

    # Import and run the Flask app directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server import app
    print("Starting server at http://0.0.0.0:5001/")
    print("Opening browser at http://127.0.0.1:5001/")
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)

if __name__ == '__main__':
    main()