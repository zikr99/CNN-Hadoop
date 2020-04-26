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
public class MaxMinInfo
{
    public int maxi;
    public double maxv;
    public int mini;
    public double minv;
    public double dv;
        
    public MaxMinInfo()
    {
        maxi = -1;
        maxv = 0.0f;
        mini = -1;
        minv = 0.0f;
        dv = 0.0f;
    }
};
