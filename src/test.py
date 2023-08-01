def userInput():
    print("Enter your multiline input. To finish, enter a blank line:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


userInput()
