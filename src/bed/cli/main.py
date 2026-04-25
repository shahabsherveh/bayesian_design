import typer

app = typer.Typer()


@app.callback()
def callback():
    """
    Awesome Portal Gun
    """
    print("Welcome to the Bayesian Experimental Design CLI!")


@app.command()
def experiment():
    print("Running experiment...")


# if __name__ == "__main__":
#     app()
