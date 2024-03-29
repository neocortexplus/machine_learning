import IPython

def main():
    # Some Python code
    list_of_numbers = [1, 2, 3, 4, 5]
    print("Before IPython embed:", list_of_numbers)
    
    # Drop into an IPython interactive shell
    print("Entering IPython interactive shell. Type exit() to resume script.")
    IPython.embed(header='You are now in an embedded IPython shell.\nYou can access variables such as list_of_numbers.\n')
    
    # Code here will run after exiting the IPython shell
    print("After IPython embed:", list_of_numbers)

if __name__ == "__main__":
    main()
