import java.io.IOException;
import java.io.PrintWriter;
import java.net.SocketTimeoutException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class Scraper {
	
	public static void main(String[] args) throws IOException, InterruptedException{
		Document d =  Jsoup.connect("https://www.winemag.com/?s=&drink_type=wine").timeout(100000).get();
		Elements ele = d.select("div.results");
		Elements list = ele.select("li.review-item");
		ArrayList<String> url = new ArrayList<String>();
		for (Element element : list) {
			url.add(element.select("a").attr("abs:href"));
			
//			System.out.println(element.select("a").attr("abs:href"));
//			System.out.println(element.text() + "\n");
		}
		
		Document d1 = null;
		for (int i = 2; i<7554;i++) {
			int j = 0;
			while (j<10) {
				try{
					d1 = Jsoup.connect("https://www.winemag.com/?s=&drink_type=wine&page=" + i).timeout(10000).get();
					break;
				}catch(SocketTimeoutException ex) {
					System.out.println(url.get(i)+ "\nTimeout after " + j + " tries.");
					TimeUnit.SECONDS.sleep(3);
				}catch(IOException e) {
					System.out.println(i);
				}
			j++;
			} 
			Elements results = d1.select("div.results");
			Elements list1 = results.select("li.review-item");
			for (Element element : list1) {
				url.add(element.select("a").attr("abs:href"));
			}
		}
		
		int size = url.size();
		ArrayList<WineReview> wineReviews = new ArrayList<WineReview>();
		ArrayList<String> faultyURL = new ArrayList<String>(); 
		for(int i = 1; i < size; i++) {
			Document d2 = null;
			boolean success = false;
			System.out.println(url.get(i));
			System.out.println(i);
//			System.out.println(url.get(i));
			if(url.get(i).contains("smartadserver") || url.get(i).isEmpty() || url.get(i).contains("paritua-2013-21%c2%b712-red-hawkes-bay"))
				continue;
			int j = 0;
			while (j<10) {
				try{
					d2 = Jsoup.connect(url.get(i)).timeout(100000).get();
					success = true;
					break;
				}catch(SocketTimeoutException ex) {
					System.out.println(url.get(i)+ "\nTimeout after " + j + " tries.");
					TimeUnit.SECONDS.sleep(3);
				}catch(IOException e) {
				}
			j++;
			}
			if(success==false) {
				faultyURL.add(url.get(i));
				continue;
			}
			
//			System.out.println(url.get(i));
			Elements reviewElements = d2.select("div.main-row");
			Elements mainReviewElements= reviewElements.select("div.large-10");
			WineReview wr = new WineReview();
			wr.setTitle(d2.select("div.article-title").text());
			wr.setPoints(d2.select("div.rating").text());
			wr.setReview(mainReviewElements.select("p.description").text());
			Elements medium9 = mainReviewElements.select("div.medium-9");
			wr.setPrice(medium9.get(0).text());
			wr.setVariety(medium9.get(1).text());
			wr.setAppellation(medium9.get(2).child(0).text());
//			wr.setProvince(medium9.get(2).child(1).text());
//			wr.setCountry(medium9.get(2).child(2).text());
			wr.setWinery(medium9.get(3).text());
			wineReviews.add(wr);
			}
		
		CSVHandler csv = new CSVHandler();
		csv.writeCSV(wineReviews);
		
		PrintWriter writer = new PrintWriter("faultyURL2.txt");
		int urlSize = faultyURL.size();
		for(int i = 0; i < urlSize; i++) {
			writer.print(faultyURL.get(i));
		}
		writer.close();
		
		}
}
