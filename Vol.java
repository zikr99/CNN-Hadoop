/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import convnet.Util;
import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class Vol implements Serializable {
    
    public int sx;
    public int sy;
    public int depth;
    public double[] w;
    public double[] dw;
    
    public Vol(int k, int g, int f)
    {
	sx = k;
	sy = g;
	depth = f;
	
        int h = k*g*f;
        
	w = Util.zeros(h);
	dw = Util.zeros(h);
    }
    
    public Vol(int k, int g, int f, double j)
    {
	sx = k;
	sy = g;
	depth = f;
	
        int h = k*g*f;
        
	w = Util.zeros(h);
	dw = Util.zeros(h);
			
	for (int d = 0; d < h; d++)
            w[d] = j;
    }
	
    public double get(int c, int g, int f)
    {
	int e = (sx*g + c)*depth + f;
	return w[e];
    }
		
    public void set(int c, int h, int g, double f)
    {
	int e = (sx*h + c)*depth + g;
	w[e] = f;
    }
		
    public void add(int c, int h, int g, double f)
    {
	int e = (sx*h + c)*depth + g;
	w[e] += f;
    }
		
    public double get_grad(int c, int g, int f)
    {
	int e = (sx*g + c)*depth + f;
	return dw[e];
    }
		
    public void set_grad(int c, int h, int g, double f)
    {
	int e = (sx*h + c)*depth + g;
	dw[e] = f;
    }
		
    public void add_grad(int c, int h, int g, double f)
    {
	int e = (sx*h + c)*depth + g;
	dw[e] += f;
    }
		
    public Vol cloneAndZero()
    {
	return new Vol(this.sx, this.sy, this.depth, 0);
    }
		
    public Vol clone()
    {
	Vol c = new Vol(this.sx, this.sy, this.depth, 0);
	int e = w.length;
			
	for (int d = 0; d < e; d++)
            c.w[d] = w[d];
			
	return c;
    }
		
    public void addFrom(Vol c)
    {
        for (int d = 0; d < w.length; d++)
            w[d] += c.w[d];
    }
		
    public void addFromScaled(Vol d, double c)
    {
	for (int e = 0; e < w.length; e++)
            w[e] += c*d.w[e];
    }
		
    public void setConst(double c)
    {
	for (int d = 0; d < w.length; d++)
            w[d] = c;
    }
};
