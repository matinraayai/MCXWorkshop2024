import argparse
from colorama import Style, Fore, Back


def pretty_print_args(args: argparse.Namespace):
    print(f"{Fore.LIGHTCYAN_EX}{Style.BRIGHT}Arguments:{Style.RESET_ALL}\n|")
    for k, v in args.__dict__.items():
        print(f"|-----\t{Fore.RED}{k}{Style.RESET_ALL}: {Fore.GREEN}{v}{Style.RESET_ALL}")
    print("")
