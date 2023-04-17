from prot2vec.tools.parsers import InterProParser
from prot2vec.tools.log import setup_logger
import logging

def run(interpro_file: str, out_file: str, clean_accessions: bool) -> None:
    log = logging.getLogger("prot2vec")
    log.info(f"Loading interpro file: {interpro_file}")
    parser = InterProParser(interpro_file, clean_accessions=clean_accessions)
    log.info("Parsing and building dataset...")
    dataset = parser.parse(ret_type="dataset")
    log.info("Writing result file...")
    dataset.to_csv(out_file, sep="\t", index=False)
    log.info("Done")

if __name__ == '__main__':
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser("Turn the output of interpro into somethingi "
                                     "that prot2vec can ingest")
    parser.add_argument("--interpro-file",
                        help="Path to the interpro file",
                        required=True)
    parser.add_argument("--output-file",
                        help="Path to the output file",
                        required=False)
    parser.add_argument("--clean-protein-accessions", 
                        help="If this flag is added, we assume the acessions are in "
                             "the UniProtKB format, and this will keep only the "
                             "accession number",
                        action="store_true")
    args = parser.parse_args()
    run(args.interpro_file, args.output_filei, args.clean_protein_accessions)
