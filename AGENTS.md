## Dev environment tips
- There is a Python virtual environment in venv; if for some reason you don't see it, create it.
- The `rg` command is installed; use it for quick searching.
- This is a MacOS environment.
- The git server we use is GitHub.
- Commit every change you make and ask the user to push changes when a significant batch of changes has been made.
- When you make a big discovery or change, note this in docs/LEARNINGS.md, even if user does not ask you to.

## Dev process tips
- Use Python logging liberally to insert judicious info, warning, and debug messages in the code.
- Import logging into each module and set logger = logging.getLogger(__name__). Then use logger for logging in that module.
- In most cases where an error message is called for, you should raise an appropriate exception. We want to know.
- The ComfyUI source code is available in ../ComfyUI for reference purposes.

## Testing instructions
- Add or update tests for the code you change, even if nobody asked.

