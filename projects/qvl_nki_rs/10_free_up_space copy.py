from pathlib import Path
from typing import Dict, List


def get_configurations() -> Dict[str, bool]:
    """
    Load configuration settings for the script.

    Returns:
        config (dict): A dictionary with configuration settings.
    """
    config = {
        "cleanup_enabled": True,
    }
    return config


def remove_target(base_path: Path, files_to_remove: List[str]) -> None:
    """
    Remove specified files from all subdirectories under a given base path.

    Args:
        base_path (Path): The base directory containing subdirectories to clean up.
        files_to_remove (List[str]): A list of filenames to delete if they exist.
    """
    for subdir in base_path.iterdir():
        if subdir.is_dir():  # Ensure it's a directory
            for filename in files_to_remove:
                file_path = subdir / filename
                if file_path.exists():
                    file_path.unlink()
                    print(f"Removed: {file_path}")


def main():
    """
    Main function to orchestrate the cleanup based on the configuration.
    """
    cfg = get_configurations()
    
    if cfg["cleanup_enabled"]:
        dir_path_3x = Path('/scratch/p290820/projects/03_nki_reader_study/output/umcg/3x')
        files_to_delete = ["rss_target.nii.gz", "rss_target_dcml.nii.gz"]
        remove_target(dir_path_3x, files_to_delete)
        
        dir_path_6x = Path('/scratch/p290820/projects/03_nki_reader_study/output/umcg/6x')
        remove_target(dir_path_6x, files_to_delete)
        
        
if __name__ == "__main__":
    main()