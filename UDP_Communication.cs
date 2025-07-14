using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System;
using System.Net.Sockets;
using System.Threading;

public class UDP_Communication : MonoBehaviour
{
    // Start is called before the first frame update
    private UdpClient udpServer;
    private IPEndPoint remoteEP;
    private Thread listenThread;

    private const int listenPort = 1904;

    void Start()
    {
        udpServer = new UdpClient(listenPort);
        remoteEP = new IPEndPoint(IPAddress.Any, 0);

        listenThread = new Thread(ListenLoop);
        listenThread.IsBackground = true;
        listenThread.Start();

        Debug.Log("UDP Ping Server gestartet auf Port " + listenPort);
    }

    void ListenLoop()
    {
        try
        {
            while (true)
            {
                // Empfangene Bytes
                byte[] receivedData = udpServer.Receive(ref remoteEP);

                // Sofort zurücksenden (Echo)
                udpServer.Send(receivedData, receivedData.Length, remoteEP);
            }
        }
        catch (SocketException ex)
        {
            Debug.Log("SocketException: " + ex.Message);
        }
        catch (Exception ex)
        {
            Debug.Log("Exception: " + ex.Message);
        }
    }

    void OnApplicationQuit()
    {
        if (listenThread != null && listenThread.IsAlive)
        {
            listenThread.Abort();
        }
        if (udpServer != null)
        {
            udpServer.Close();
        }
    }
}
