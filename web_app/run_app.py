import webbrowser
import time
import threading
from server import app

def open_browser():
    """Open the web browser after a delay"""
    # Wait a bit for the server to start
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:5001')

def main():
    # Start the browser opening in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    main()