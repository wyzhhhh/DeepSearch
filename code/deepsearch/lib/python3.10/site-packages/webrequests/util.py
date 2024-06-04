from pathlib import Path

import tqdm


def safe_open(filename, mode='r'):
    file = Path(filename)

    if 'w' in mode and not file.parent.exists():
        file.parent.mkdir(parents=True)

    if filename.endswith('.gz'):
        import gzip
        return gzip.open(filename, mode=mode)

    return file.open(mode=mode)


def file_download_bar(total, desc='', colour='green'):
    """generate file download progress bar
    """
    bar = tqdm.tqdm(total=total,
                    desc=desc,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    colour=colour)
    return bar
