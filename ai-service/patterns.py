# Technical support conversation patterns and dictionaries

FOLLOW_UP_PATTERNS = {
    'confirmation': [
        "yes", "yeah", "yep", "ok", "okay", "alright", "sure", "correct",
        "no", "nope", "not", "didn't", "doesnt", "still", "right",
        "exactly", "precisely", "indeed", "absolutely", "definitely",
        "nah", "negative", "wrong", "incorrect", "never", "disagree",
        "agree", "fine", "good", "perfect", "great", "understood",
        "got it", "i see", "makes sense", "clear", "that's right",
        "that's wrong", "not quite", "almost", "kind of", "sort of",
        "maybe", "perhaps", "possibly", "probably", "actually", "well",
        "hmm", "uh", "um", "eh", "meh", "dunno", "don't know"
    ],
    'clarification': [
        "what about", "how about", "what if", "but", "and", "or",
        "then", "so", "because", "why", "where", "when", "how",
        "could you", "would you", "can you", "please explain",
        "don't understand", "confused", "unclear", "not sure",
        "meaning", "meant", "mean", "specifically", "exactly",
        "detail", "more info", "elaborate", "clarify", "specify",
        "example", "instance", "scenario", "case", "situation",
        "context", "reference", "regarding", "about", "concerning",
        "related to", "with respect to", "in terms of", "speaking of",
        "on that note", "by the way", "meanwhile", "in addition",
        "furthermore", "moreover", "besides", "apart from", "except",
        "other than", "rather than", "instead of", "alternatively"
    ],
    'additional_info': [
        "also", "additionally", "moreover", "plus", "further",
        "another", "more", "else", "other", "extra", "additional",
        "next", "following", "subsequent", "furthermore", "besides",
        "along with", "together with", "coupled with", "combined with",
        "in addition to", "as well as", "not to mention", "including",
        "included", "apart from", "aside from", "other than", "except",
        "excepting", "excluding", "save for", "but for", "besides",
        "furthermore", "moreover", "too", "similarly", "likewise",
        "equally", "correspondingly", "in the same way", "by the same token",
        "in like manner", "analogously", "comparatively", "add", "added"
    ],
    'problem_persistence': [
        "still", "again", "same", "persist", "continues", "ongoing",
        "not working", "didn't work", "doesn't work", "won't work",
        "failed", "failing", "keeps", "continuing", "constant", "constantly",
        "repeatedly", "repetitive", "recurring", "recurrent", "persistent",
        "won't stop", "doesn't stop", "never stops", "all the time",
        "frequently", "regular", "regularly", "consistently", "continuous",
        "continually", "perpetually", "eternally", "endlessly", "always",
        "every time", "each time", "time and again", "over and over",
        "repeatedly", "keeps happening", "keeps occurring", "keeps showing",
        "remains", "remaining", "stayed", "staying", "lingering", "lasting",
        "persisting", "maintained", "maintained", "sustained", "prolonged"
    ]
}

IMPORTANT_KEYWORDS = {
    # Network related
    'wireless', 'network', 'wifi', 'connection', 'internet', 
    'router', 'modem', 'signal', 'connectivity',
    # Hardware related
    'hardware', 'device', 'computer', 'laptop', 'desktop',
    'screen', 'display', 'keyboard', 'mouse', 'battery',
    'power', 'charging', 'usb', 'port', 'cable',
    # Software related
    'software', 'program', 'application', 'app', 'windows',
    'mac', 'update', 'install', 'driver', 'system',
    # Performance related
    'slow', 'fast', 'speed', 'performance', 'memory',
    'ram', 'cpu', 'processor', 'disk', 'storage',
    # Error related
    'error', 'issue', 'problem', 'fail', 'crash',
    'freeze', 'hang', 'stop', 'break', 'bug',
    # Status words
    'not', 'working', 'broken', 'failed', 'dead',
    'stuck', 'frozen', 'crashed', 'slow', 'overheating'
}

TECH_CATEGORIES = {
    'wifi': [
        'wifi', 'wireless', 'router', 'network', 'connection', 'internet', 
        'wifi', 'wi-fi', 'wiifi', 'wirless', 'ruter', 'conection', 'inet',
        'broadband', 'ethernet', 'lan', 'wan', 'network adapter', 'access point',
        'hotspot', 'connectivity', 'gateway', 'ip address', 'dns', 'dhcp',
        'subnet', 'proxy', 'vpn', 'firewall', 'protocol', 'bandwidth', 'ping',
        'latency', 'packet loss', 'interference', 'channel', 'ssid', 'mesh',
        'bridge mode', 'port forwarding', 'nat', 'qos', 'dual band'
    ],
    # ... (rest of the categories)
}

MISSPELLINGS = {
    # Network-related
    'overhiting': 'overheating', 'overheat': 'overheating',
    'overheated': 'overheating', 'heating': 'overheating',
    'hot': 'overheating', 'temperature': 'overheating',
    # ... (rest of the misspellings)
}

PROBLEM_INDICATORS = [
    'issue', 'problem', 'error', 'not working', 'failed', 'help',
    'isue', 'problm', 'eror', 'notworking', 'faild', 'halp', 'hlp',
    'trouble', 'truble', 'fix', 'broken', 'brokn', 'stuck', 'stuk'
]

QUESTION_INDICATORS = [
    'how', 'what', 'why', 'where', 'when', 'can', 'could',
    'hw', 'wut', 'wy', 'wher', 'wen', 'cn', 'cud'
]

TROUBLESHOOTING_KEYWORDS = [
    'error', 'eror', 'err', 'issue', 'isue', 'prob',
    'problem', 'problm', 'trouble', 'fix', 'help', 'halp',
    'not working', 'notworking', 'broken', 'failed', 'faild', 'stuck',
    'crash', 'crashd', 'bug', 'debug', 'cant', 'cannot',
    'wont', 'doesnt', 'frozen', 'slow', 'connection', 'conectn'
] 