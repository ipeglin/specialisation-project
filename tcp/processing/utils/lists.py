# from https://stackoverflow.com/a/312464 under licence CC BY-SA 4.0
# By: Ned Batchelder
# Edited by: Mateen Ulhaq & Ian Philip Eglin
# Edits made: Changed parameter names to be more descriptive
def chunks(lst, num_chunks):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), num_chunks):
       yield lst[i:i + num_chunks] 