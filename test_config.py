from Utils import Configuration
from rich import print as rprint

def run():
    config = Configuration.load_run("./run.ini")
    rprint(config)

if __name__ == "__main__":
    run()