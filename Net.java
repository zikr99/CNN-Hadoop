/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

/**
 *
 * @author zikr
 */

import java.io.*;
import java.util.List;
import java.util.ArrayList;
import convnet.Util;
import convnet.Vol;
import convnet.*;


public class Net implements Serializable {

    /**
     * @param args the command line arguments
     */
    
    public List<Layer> layers;
    
    public void makeLayers()
    {
        layers = new ArrayList<Layer>();
        
        InputLayer inp = new InputLayer(24, 24, 1);
        ConvLayer cv1 = new ConvLayer(24, 24, 1, 5, 5, 1, 2, 8, 1.0f, 0.001f, 0.0f);
        ReluLayer rl1 = new ReluLayer(cv1);
        PoolLayer pl1 = new PoolLayer(rl1.out_sx, rl1.out_sy, rl1.out_depth, 2, 2, 2, 0);
        ConvLayer cv2 = new ConvLayer(pl1.out_sx, pl1.out_sy, pl1.out_depth, 5, 5, 1, 2, 16, 1.0f, 0.001f, 0.0f);
        ReluLayer rl2 = new ReluLayer(cv2);
        PoolLayer pl2 = new PoolLayer(rl2.out_sx, rl2.out_sy, rl2.out_depth, 3, 3, 3, 0);
        FullyConnLayer fc1 = new FullyConnLayer(pl2.out_sx, pl2.out_sy, pl2.out_depth, 10, 1.0f, 0.001f, 0.0f);
        SoftmaxLayer sm1 = new SoftmaxLayer(1, 1, fc1.out_depth);
        
        layers.add(inp);
        layers.add(cv1);
        layers.add(rl1);
        layers.add(pl1);
        layers.add(cv2);
        layers.add(rl2);
        layers.add(pl2);
        layers.add(fc1);
        layers.add(sm1);
    }
    
    public Vol forward(Vol f, int h)
    {
        Vol e = layers.get(0).forward(f, h);
			
	for (int g = 1; g < layers.size(); g++)
            e = layers.get(g).forward(e, h);
			
	return e;
    }
    
    public double backward(int h)
    {
        int g = layers.size();
	double f = ((FinalLayer)layers.get(g - 1)).backward(h);
			
	for (int e = g - 2; e >= 0; e--)
            layers.get(e).backward();
			
	return f;
    }
    
    public double getCostLoss(Vol e, int h)
    {
        forward(e, 0);
			
	int g = layers.size();
	double f = ((FinalLayer)layers.get(g - 1)).backward(h);
	return f;
    }
    
    public int getPrediction()
    {
	Layer h = this.layers.get(this.layers.size() - 1);
		
	double[] j = h.out_act.w;
	double e = j[0];
        int f = 0;

	for (int g = 1; g < j.length; g++)
	{
            if (j[g] > e)
            {
		e = j[g];
		f = g;
            }
	}
			
	return f;
    }
    
    public List<ParamsGrads> getParamsAndGrads()
    {
        List<ParamsGrads> e = new ArrayList<ParamsGrads>();
        
        for (int g = 0; g < this.layers.size(); g++)
        {
            List<ParamsGrads> h = ((Layer)this.layers.get(g)).getParamsAndGrads();

            for (int f = 0; f < h.size(); f++)
            {
		e.add(h.get(f));
            }
	}
			
	return e;
    }
    
    public void write()
    {
        
    }
    
    public void read()
    {
    
    }
};
