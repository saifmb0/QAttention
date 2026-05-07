FILE_NAME=paper
while true; do
  echo "$(date) Waiting for save..."
  # This waits for the file to be closed after a write 
  inotifywait -q -e close_write "$FILE_NAME.tex"

  echo "Compiling..."
  pdflatex -interaction=nonstopmode "$FILE_NAME.tex"
  pdflatex -interaction=nonstopmode "$FILE_NAME.tex"
  # Clean up temp files
  rm -f "$FILE_NAME.log" "$FILE_NAME.aux" "$FILE_NAME.out"

  echo "Done iteration. Monitoring again..."
done
