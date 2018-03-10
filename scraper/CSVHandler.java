import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import au.com.bytecode.opencsv.CSVWriter;

public class CSVHandler {
	
	public void writeCSV(ArrayList<WineReview> wr) {
		
		try {
			CSVWriter writer = new CSVWriter(new FileWriter("WineReviews2.csv"));
			String[] output = "Title,Rating,Review,Price,Variety,Appellation,Winery".split(",");
			writer.writeNext(output);
			int size = wr.size();
			for(int i = 0; i<size;i++) {
				String[] writeOutput = {wr.get(i).getTitle(),wr.get(i).getPoints(),wr.get(i).getReview(),wr.get(i).getPrice(),
						wr.get(i).getVariety(),wr.get(i).getAppellation(),wr.get(i).getWinery()};
				writer.writeNext(writeOutput);
			}
			writer.close();
		}
		catch (IOException ioe)
		{
			System.out.println(ioe.getMessage() + "Error reading file");
		}
	}
}
