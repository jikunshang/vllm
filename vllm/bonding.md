# how to make 8 netcards to 1 bonding card

1. install ifenslave and net-tools

```
 sudo apt install ifenslave net-tools -y
```

2. down all network interface

```
sudo ifconfig ens108np0 down
sudo ifconfig ens109np0 down
sudo ifconfig ens110np0 down
sudo ifconfig ens111np0 down
sudo ifconfig ens112np0 down
sudo ifconfig ens113np0 down
sudo ifconfig ens114np0 down
sudo ifconfig ens115np0 down
```

3. create bond and setting

```
sudo ip link add bond0 type bond mode 802.3ad

sudo ip link set ens108np0 master bond0
sudo ip link set ens109np0 master bond0
sudo ip link set ens110np0 master bond0
sudo ip link set ens111np0 master bond0
sudo ip link set ens112np0 master bond0
sudo ip link set ens113np0 master bond0
sudo ip link set ens114np0 master bond0
sudo ip link set ens115np0 master bond0

sudo ifconfig bond0 up
sudo ip link

```

4. configure /etc/netplan/01-network-manager-all.yaml
```
# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager

  ethernets:
    ens108np0:
      dhcp4: no
    ens109np0:
      dhcp4: no
    ens110np0:
      dhcp4: no
    ens111np0:
      dhcp4: no
    ens112np0:
      dhcp4: no
    ens113np0:
      dhcp4: no
    ens114np0:
      dhcp4: no
    ens115np0:
      dhcp4: no

  bonds:
      bond0:
       interfaces: [ens108np0, ens109np0, ens110np0, ens111np0, ens112np0, ens113np0, ens114np0, ens115np0]
       addresses: [192.168.1.130/24]
       routes:
         - to: default
       parameters:
         mode: balance-rr
         transmit-hash-policy: layer3+4
         mii-monitor-interval: 1

```

5. down interface again(I don't why, but the guide told me do that)

```
sudo ifconfig ens108np0 down
sudo ifconfig ens109np0 down
sudo ifconfig ens110np0 down
sudo ifconfig ens111np0 down
sudo ifconfig ens112np0 down
sudo ifconfig ens113np0 down
sudo ifconfig ens114np0 down
sudo ifconfig ens115np0 down
```

6. apply net plan and bring bond0 up

```
sudo netplan apply
sudo ifconfig bond0 up
ifconfig bond0
```