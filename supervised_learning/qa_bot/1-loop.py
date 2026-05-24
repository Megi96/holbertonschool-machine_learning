#!/usr/bin/env python3
"""Interactive question-answer loop."""


def qa_loop():
    """
    Starts an interactive loop that accepts user questions.

    The program exits when the user enters:
    exit, quit, goodbye, or bye (case insensitive).
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        print('A:')


if __name__ == '__main__':
    qa_loop()
