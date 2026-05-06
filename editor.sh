FILE_NAME=experiment_figures.py
while true; do
  echo "$(date) Waiting for save..."
  # This waits for the file to be closed after a write 
  inotifywait -q -e close_write "$FILE_NAME"

  echo "Compiling..."
  python3 $FILE_NAME
  # Clean up temp files
  

  echo "Done iteration. Monitoring again..."
done
