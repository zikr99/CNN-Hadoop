/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.util.*;
import convnet.MaxMinInfo;

/**
 *
 * @author zikr
 */

public class Util
{
    private static boolean k = false;
    private static double e = 0;
	
    protected static double l()
    {
        if (k)
        {
            k = false;
            return e;
        }
		
        double q = 2*Math.random() - 1;
	double p = 2*Math.random() - 1;
	double s = q*q + p*p;
		
	if ((s == 0) || (s > 1))
	{
            return l();
        }
		
	double t = Math.sqrt(-2*Math.log(s)/s);
	e = p*t;
	k = true;

        return q*t;
    }
	
    public static double randf(double q, double p)
    {
        return Math.random()*(p - q) + q;
    }
		
    public static int randi(int q, int p)
    {
        return (int)Math.floor(Math.random()*(p - q) + q);
    }
			
    public static int randn(int q, int p)
    {
        return (int)Math.round(q + l()*p);
    }
		
    public static double[] zeros(int r)
    {	
        if (r > 0)
	{
            double[] p = new double [r];
	
            for (int q = 0; q < r; q++)
		p[q] = 0;
				
            return p;
	}
	else
            return new double [0];
    }
	
    public static boolean arrContains(double[] p, double q)
    {
        for (int r = 0, s = p.length; r < s; r++)
	{
            if (p[r] == q)
		return true;
	}
		
        return false;
    }
    
    public static boolean arrContains2(double[] p, double q, int num)
    {
        for (int r = 0; r < num; r++)
	{
            if (p[r] == q)
		return true;
	}
		
        return false;
    }
	
    public static double[] arrUnique(double[] q)
    {
        double[] p = new double [q.length];
        int num = 0;
		
	for (int r = 0, s = q.length; r < s; r++)
            if (!arrContains2(p, q[r], num))
            {
                num++;
		p[num - 1] = q[r];
            }
        
        double[] v = new double [num];
	
        for (int r = 0; r < num; r++)
            v[r] = p[r];
        
	return v;
    }
    
    public static MaxMinInfo maxmin(double[] q)
    {
        MaxMinInfo mm = new MaxMinInfo();
        
	if (q.length == 0)
            return mm;
		
	double p = q[0];
	double s = q[0];
	int r = 0;
	int u = 0;
	int v = q.length;
		
	for (int t = 1; t < v; t++)
	{
            if (q[t] > p)
            {
		p = q[t];
		r = t;
            }
		
            if (q[t] < s)
            {
		s = q[t];
		u = t;
            }
	}
	
        mm.maxi = r;
        mm.maxv = p;
        mm.mini = u;
        mm.minv = s;
        mm.dv = p - s;
        
	return mm;
    }

    public static int[] randperm(int v)
    {
        int s = v, r = 0, p;
	int[] u = new int [v];
		
	for (int t = 0; t < v; t++)
            u[t] = t;
		
	while (s > 0)
	{
            r = (int)Math.floor(Math.random()*(s + 1));
            p = u[s];
            u[s] = u[r];
            u[r] = p;
                                
            s--;
	}
		
	return u;
    }
	
    public static double weightedSample(double[] q, double[] v)
    {
	double t = randf(0.0f, 1.0f);
	double s = 0;
		
	for (int r = 0, u = q.length; r < u; r++)
	{
            s += v[r];
			
            if (t < s)
		return q[r];
	}
        
        return q[q.length - 1];
    }
};
