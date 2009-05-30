/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 INESC Porto
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */

#ifndef TAP_NET_DEVICE_H
#define TAP_NET_DEVICE_H

#include "ns3/address.h"
#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/traced-callback.h"

namespace ns3 {


/**
 * \class TapNetDevice
 * \brief A virtual device, similar to Linux TAP interfaces.
 *
 * A TapNetDevice is a "virtual" NetDevice implementation which
 * delegates to a user callback (see method SetSendFromCallback()) the
 * task of actually transmitting a packet.  It also allows the user
 * code to inject the packet as if it had been received by the
 * TapNetDevice.  Together, these features allow one to build tunnels.
 * For instance, by transmitting packets into a UDP socket we end up
 * building an IP-over-UDP-over-IP tunnel.
 *
 * The same thing could be accomplished by subclassing NetDevice
 * directly.  However, TapNetDevice is usually much simpler to program
 * than a NetDevice subclass.
 */
class TapNetDevice : public NetDevice
{
public:
  typedef Callback<bool, Ptr<Packet>, const Address&, const Address&, uint16_t> SendFromCallback;

  static TypeId GetTypeId (void);
  TapNetDevice ();

  virtual ~TapNetDevice ();

  /**
   * \brief Set the user callback to be called when a L2 packet is to be transmitted
   * \param transmitCb the new transmit callback
   */
  void SetSendFromCallback (SendFromCallback transmitCb);

  /**
   * \brief Configure whether the virtual device needs ARP
   *
   * \param needsArp the the 'needs arp' value that will be returned
   * by the NeedsArp() method.  The method IsBroadcast() will also
   * return this value.
   */
  void SetNeedsArp (bool needsArp);

  /**
   * \brief Configure whether the virtual device supports SendFrom
   */
  void SetSupportsSendFrom (bool supportsSendFrom);

  /**
   * \brief Configure the reported MTU for the virtual device. The
   * default value is 65535.
   * \return whether the MTU value was within legal bounds
   */
  bool SetMtu (const uint16_t mtu);


  /**
   * \param p packet sent from below up to Network Device
   * \param protocol Protocol type
   * \param source the address of the sender of this packet.
   * \returns true if the packet was forwarded successfully,
   *          false otherwise.
   *
   * Forward a "virtually received" packet up the node's protocol
   * stack.
   */
  bool Receive (Ptr<Packet> packet, uint16_t protocol, const Address &source);


  /**
   * \param p packet sent from below up to Network Device
   * \param protocol Protocol type
   * \param source the address of the sender of this packet.
   * \param destination the address of the receiver of this packet.
   * \param packetType type of packet received (broadcast/multicast/unicast/otherhost)
   * \returns true if the packet was forwarded successfully, false otherwise.
   *
   * Forward a "virtually received (in promiscuous mode)" packet up
   * the node's protocol stack.
   */
  bool PromiscReceive (Ptr<Packet> packet, uint16_t protocol,
                       const Address &source, const Address &destination,
                       PacketType packetType);


  // inherited from NetDevice base class.
  virtual void SetIfIndex(const uint32_t index);
  virtual uint32_t GetIfIndex(void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  virtual Address GetAddress (void) const;
  virtual uint16_t GetMtu (void) const;
  virtual bool IsLinkUp (void) const;
  virtual void SetLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast (void) const;
  virtual Address GetBroadcast (void) const;
  virtual bool IsMulticast (void) const;
  virtual Address GetMulticast (Ipv4Address multicastGroup) const;
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom () const;
  virtual bool IsBridge (void) const;

protected:

  virtual void DoDispose (void);

private:

  SendFromCallback m_sendCb;
  TracedCallback<Ptr<const Packet> > m_rxTrace;
  TracedCallback<Ptr<const Packet> > m_txTrace;
  Ptr<Node> m_node;
  ReceiveCallback m_rxCallback;
  PromiscReceiveCallback m_promiscRxCallback;
  std::string m_name;
  uint32_t m_index;
  uint16_t m_mtu;
  bool m_needsArp;
  bool m_supportsSendFrom;
};

}; // namespace ns3

#endif

