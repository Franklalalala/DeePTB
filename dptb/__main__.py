from dptb.entrypoints.main import main as entry_main
import logging
from dptb import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def print_logo():
    import pyfiglet

    f = pyfiglet.Figlet(font='dos_rebel')
    logo = f.renderText("DeePTB")
    log.info(" ")
    log.info(" ")
    log.info("#"*81)
    log.info("#" + " "*79 + "#")
    log.info("#" + " "*79 + "#")
    for line in logo.split('\n'):
        if line.strip():
            log.info('#     '+line+ '     #')
    log.info("#" + " "*79 + "#")
    version_info = f"Version: {__version__}"
    padding = (79 - len(version_info)) // 2
    nspace = 79-padding
    format_str = "#" + "{}"+"{:<"+f"{nspace}" + "}"+ "#"         
    log.info(format_str.format(" "*padding, version_info))
    log.info("#" + " "*79 + "#")
    log.info("#"*81)
    log.info(" ")
    log.info(" ")
def main(argv=None):
    """
    The main entry point for the dptb package.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in {'h0', '-h', '--help', '-v', '--version'}:
        print_logo()
    return entry_main(args)

if __name__ == '__main__':
    raise SystemExit(main())
