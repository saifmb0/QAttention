while true;  do
  echo "$(date) Waiting for save..."
  # This waits for the file to be closed after a write
  inotifywait -q -e close_write "./paper.tex"

  echo "Compiling..."
  pdflatex -interaction=nonstopmode ./paper.tex

  # Clean up temp files
  rm -f redacted.log redacted.aux redacted.out

  echo "Done iteration. Monitoring again..."
done
