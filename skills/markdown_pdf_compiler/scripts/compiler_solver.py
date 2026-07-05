import os
import subprocess

def compile_markdown_to_pdf(input_path: str, output_path: str, font: str = 'Liberation Sans', working_dir: str = None) -> dict:
    """
    Compiles a Markdown (.md) file to a PDF (.pdf) file using Pandoc and the Typst PDF engine,
    configuring the fallback font to bypass empty system font fallback errors on WSL/Nix.

    Args:
        input_path: Absolute path to the input Markdown file.
        output_path: Absolute path to the destination PDF file.
        font: Font name to pass to pandoc's mainfont variable. Defaults to 'Liberation Sans'.

        working_dir: The directory from which relative asset paths (like images) are resolved.
                     If None, it defaults to the directory of the input file.

    Returns:
        dict: A dictionary containing 'status' ('success' or 'error'), and 'message' or 'error_output'.
    """
    if not os.path.exists(input_path):
        return {
            "status": "error",
            "error_output": f"Input file not found at: {input_path}"
        }

    # Resolve default working directory to the input file's parent dir
    if not working_dir:
        working_dir = os.path.dirname(input_path)

    # Prepare command
    cmd = [
        "pandoc",
        input_path,
        "-o",
        output_path,
        "--pdf-engine=typst",
        "-V",
        f"mainfont={font}"
    ]

    # Setup environment variables
    # We do NOT ignore system fonts completely here, but we can set TYPST_FONT_PATHS
    # if necessary, or just run with normal environments.
    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "message": f"Successfully compiled {os.path.basename(input_path)} to {os.path.basename(output_path)}."
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "error_output": f"Pandoc compile failed (exit code: {e.returncode}). Stderr: {e.stderr}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_output": f"An unexpected error occurred: {str(e)}"
        }
