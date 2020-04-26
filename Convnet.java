/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.io.*;
import java.util.List;
import java.util.ArrayList;
import convnet.Util;
import convnet.Vol;
import convnet.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

/**
 *
 * @author zikr
 */
public class Convnet {

    /**
     * @param args the command line arguments
     */
    
    private static int[] getPixelData(BufferedImage img, int x, int y) 
    {
        int argb = img.getRGB(x, y);

        int rgb[] = new int[] {
            (argb >> 16) & 0xff, //red
            (argb >>  8) & 0xff, //green
            (argb      ) & 0xff  //blue
        };
                
        return rgb;
    }
    
    public static void main(String[] args) {
        // TODO code application logic here
        
        int n_train = 200;
        int n_val = 100;
        
        List<Vol> data = new ArrayList<Vol>();
        double[] labels = new double [n_train];
        List<Vol> valdata = new ArrayList<Vol>();
        double[] vallabels = new double [n_val];
        
        String line = null;
        
        try 
        {
            FileReader fileReader = new FileReader("D:/MNIST/Training/labels.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            for (int i = 1; i <= n_train + n_val; i++) 
            {
                line = bufferedReader.readLine();
                
                if (i <= n_train)
                    labels[i - 1] = Double.parseDouble(line);
                else
                    vallabels[i - n_train - 1] = Double.parseDouble(line);
                    
                String fname = "D:/MNIST/Training/" + String.format("%06d.jpg", i);
                System.out.println(fname);
                
                BufferedImage img = null;
                img = ImageIO.read(new File(fname));
                
                int[] rgb;
                Vol inp = new Vol(img.getWidth(), img.getHeight(), 1);
                
                for (int m = 0; m < img.getHeight(); m++)
                {
                    for (int n = 0; n < img.getWidth(); n++)
                    {
                        rgb = getPixelData(img, m, n);
                        inp.set(n, m, 0, (double)rgb[0]);
                    }
                }    
                
                if (i <= n_train)
                    data.add(inp);
                else
                    valdata.add(inp);
            }      

            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) 
        {    
        }
        catch(IOException ex) 
        {
        }
        
        Net net = new Net();
        net.makeLayers();
        
        Trainer trainer = new Trainer(net, "adagrad", 0.01, 0.0, 50, 
            0.001, 0.001, 0.95, 0.000001);
        
        trainer.train(data, labels, valdata, vallabels, 100);
        
        try 
        {
            FileOutputStream fos = new FileOutputStream("net.ser");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(net);
            oos.close();
	} 
        catch (FileNotFoundException e) 
        {
            e.printStackTrace();
	} 
        catch (IOException e) 
        {
            e.printStackTrace();
	}
    }
};
