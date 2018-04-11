# AI Chat
AI Based Chatbot

Requirements
Python >= 2.7
Flask >= 0.10

## Installation
Install **Flask** framework and **AIML** module.
    ```bash
    pip install Flask
    pip install aiml
    ```
Run the python server.
    ```bash
    python main.py
    ```
Open **http://127.0.0.1:5000** in your browser.
You're done and let's chat with ALICE.





site-packages/mpld3/urls.py
"""
mpld3 URLs
==========
URLs and filepaths for the mpld3 javascript libraries
"""

import os
from . import __path__, __version__
import warnings

__all__ = ["D3_URL", "MPLD3_URL", "MPLD3MIN_URL",
           "D3_LOCAL", "MPLD3_LOCAL", "MPLD3MIN_LOCAL"]

WWW_JS_DIR = "https://mpld3.github.io/js/"
D3_URL =  "/js/dummy.js"
MPLD3_URL = WWW_JS_DIR + "mpld3.v{0}.js".format(__version__)
MPLD3MIN_URL = WWW_JS_DIR + "mpld3.v{0}.min.js".format(__version__)

LOCAL_JS_DIR = os.path.join(__path__[0], "js")
D3_LOCAL = os.path.join(LOCAL_JS_DIR, "d3.v3.min.js")
MPLD3_LOCAL = os.path.join(LOCAL_JS_DIR,
                           "mpld3.v{0}.js".format(__version__))
MPLD3MIN_LOCAL = os.path.join(LOCAL_JS_DIR,
                              "mpld3.v{0}.min.js".format(__version__))




site-packages/mpld3/_display.py
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
            numpy.uint16,numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
            numpy.float64)):
            return float(obj)
        elif isinstance(obj,(numpy.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

site-packages/pyLDAvis/_display.py

    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("{{ d3_url }}", function(){
         LDAvis_load_lib("{{ ldavis_url }}", function(){
                 new LDAvis("#" + {{ visid }}, {{ visid_raw }}_data);
            })
         });

will become

    // require.js not available: dynamically load d3 & LDAvis
    
         LDAvis_load_lib("{{ ldavis_url }}", function(){
                 new LDAvis("#" + {{ visid }}, {{ visid_raw }}_data);
            })



