
outdir="D:\on2\cosmic2_gis"
mkdir -p "$outdir"

failed_log="${outdir}/failed_downloads.txt"

# clear old log if it exists
> "$failed_log"

for year in $(seq 2019 2027); do
  for day in $(seq 1 366); do
    if [ "$year" -eq 2019 ] && [ "$day" -lt 231 ]; then
      continue
    fi

    for hour in $(seq 0 23); do
      # zero-pad day to three digits
      hr=$(printf "%02d" "$hour")
      ddd=$(printf "%03d" "$day")
      url="https://tacc.cwa.gov.tw/data-service/fs7rt_trops/level3/GIS/${year}.${ddd}/GIS_Ne_IRI_RO_GPS_${year}_${ddd}_${hr}.nc"
      outfile="${outdir}/GIS_Ne_IRI_RO_GPS_${year}_${ddd}_${hr}.nc"
      echo "Downloading $url"

      if [ -f "$outfile" ]; then
          echo "File ${file} already exists. Skipping."
      else
          curl -L -o "$outfile" "$url"
      fi

      if [ $? -ne 0 ]; then
        echo "Failed: $url"
        echo "$url" >> "$failed_log"
        # optional: remove empty file
        [ -f "$outfile" ] && [ ! -s "$outfile" ] && rm "$outfile"
      fi

    done 
  done 
done


echo "All downloads attempted. Failed URLs saved to $failed_log"
