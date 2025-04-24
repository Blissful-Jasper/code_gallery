# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

@email : xianpuji@hhu.edu.cn
"""
from math import pi, acos, sqrt,floor, ceil
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fft
from joblib import Parallel, delayed
# ================================================================================================
# Author: %(Jianpu)s | Affiliation: Hohai
# email : xianpuji@hhu.edu.cn
# Last modified:  %(date)s
# Filename: 
# =================================================================================================


def extract_low_harmonics(data: xr.DataArray, n_harm: int = 3, dim: str = 'dayofyear') -> xr.DataArray:
    """
    从逐日气候态中提取指定阶数的谐波并重构信号。

    参数：
    - data: 输入的xarray.DataArray（时间维度应为dayofyear的气候态）。
    - n_harm: 要保留的最高谐波阶数（保留 0~n_harm-1 的谐波，第 n_harm 的系数减半）。
    - dim: 要进行 FFT 的维度（默认是 'dayofyear'）。

    返回：
    - 仅包含低阶谐波的重建数据，类型为 xarray.DataArray。
    """
    # 傅里叶变换
    z_fft = np.fft.rfft(data, axis=data.get_axis_num(dim))
    # 设置频率
    freqs = np.fft.rfftfreq(data.sizes[dim])
    # print(1/freqs[1:])
    # 保留低阶谐波并处理第 n_harm 阶的振幅
    z_fft_n = z_fft.copy()
 
    z_fft_n[n_harm,:,:] *= 0.5  # 第 n_harm 阶振幅减半
    z_fft_n[(n_harm+1):,:,:] = 0
  
    # 反傅里叶变换，保留实数部分
    clim_low_harm = np.fft.irfft(z_fft_n, n=data.sizes[dim], axis=data.get_axis_num(dim)).real
    
    # 保持 xarray 格式和原数据一致
    coords = {k: v for k, v in data.coords.items()}
    dims = data.dims
    attrs = {
        "smoothing"     :   f"FFT: {n_harm} harmonics were retained.",
        "information"   :   "Smoothed daily climatological averages",
        "units"         :   "W/m^2",
        "long_name"     :   "Daily Climatology: Daily Mean OLR",
        }
    
    return xr.DataArray(clim_low_harm, coords=coords, dims=dims, attrs=attrs)


    
 


def check_filter_wave(python_result,ncl_path,wave_name):
    
    ds_ncl = xr.open_dataset(ncl_path)
    clm_ncl = ds_ncl[wave_name]
    
    random_index = np.random.randint(0, 7305)
    # print(clm_ncl.time[random_index])
    # 对比用的纬向平均或空间平均
    clm_ncl_mean = clm_ncl[:random_index].std(['lon','lat'])
    clim_py_mean = python_result[:random_index].std(['lon','lat'])
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Python vs NCL Harmonics Climatology Comparison", fontsize=16)
    # 1. 平均气候态曲线对比
    ax = axes[0, 0]
    clim_py_mean.plot(ax=ax, label='Python')
    clm_ncl_mean.plot(ax=ax, label='NCL')
    ax.set_title('(a) Zonal Mean Climatology Curve')
    ax.legend()
    
    # 2. Python 版本空间图（均值沿 dayofyear）
    ax = axes[0, 1]
    python_result.std('time').plot.contourf(ax=ax, cmap='jet',levels=21,extend='neither')
    ax.set_title('(b) Python: STD Climatology')
    
    # 3. NCL 版本空间图（均值沿 year_day）
    ax = axes[1, 0]
    clm_ncl.std('time').plot.contourf(ax=ax, cmap='jet',levels=21,extend='neither')
    ax.set_title('(c) NCL: STD Climatology')
    
    # 4. 差值图（Python - NCL）
    ax = axes[1, 1]
    (python_result.std('time') - clm_ncl.std('time')).plot.contourf(ax=ax, cmap='RdYlBu')
    ax.set_title('(d) Difference: Python - NCL')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show() 
    



def kf_filter(in_data, lon, obs_per_day, t_min, t_max, k_min, k_max, h_min, h_max, wave_name):
    """
    Apply WK99 Kelvin wave filter to 2D time-longitude data.
    
    Parameters:
        in_data: xarray.DataArray with dims ("time", "lon")
        obs_per_day: time resolution (e.g., 4 for 6-hourly)
        t_min, t_max: filtering period range (in days)
        k_min, k_max: wavenumber range
        h_min, h_max: equivalent depth range
        wave_name: name of the wave type
    Returns:
        filtered: xarray.DataArray with same shape as input
    """
    is_xarray = isinstance(in_data, xr.DataArray)

    if is_xarray:
        data_np = in_data.values
        time_dim, lon_dim = in_data.sizes["time"], in_data.sizes["lon"]
    else:
        data_np = in_data
        time_dim, lon_dim = data_np.shape
        
    
    
    wrap_flag = np.isclose((lon[0] + 360) % 360, lon[-1] % 360)

    if wrap_flag:
        data = in_data.isel(lon=slice(1, None))  # 丢掉第一个点
    else:
        data = in_data
    
    # Detrend and taper
    data_np = signal.detrend(data,axis=0)
    
    data_np = signal.windows.tukey(time_dim, alpha=0.05)[:,np.newaxis]*(data_np)

    # FFT now shape is: timexlon
    fft_data = fft.rfft2(data_np,axes=(1,0))
    fft_data[:, 1:] = fft_data[:, -1:0:-1]

    # Frequency/wavenumber axis : Find the indeces for the period cut-offs
    freq_dim = fft_data.shape[0]
    k_dim = fft_data.shape[1]
    j_min = int(time_dim / (t_max * obs_per_day))
    j_max = int(time_dim / (t_min * obs_per_day))
    j_max = min(j_max, freq_dim)
    #  Find the indeces for the wavenumber cut-offs
    #  This is more complicated because east and west are separate
    if k_min < 0:
        i_min = max(k_dim + k_min, k_dim // 2)
    else:
        i_min = min(k_min, k_dim // 2)
    if k_max < 0:
        i_max = max(k_dim + k_max, k_dim // 2)
    else:
        i_max = min(k_max, k_dim // 2)

    # Bandpass filter by frequency: set the appropriate coefficients to zero
    # here j_min=365, j_max=2435, set :365==0 and 2435:==0
    # same i_min=2 ,  i_max=14,   set :2==0   and 14:==0
    if j_min>0:
        
        fft_data[:j_min-1,:] = 0
        
    if j_max<freq_dim-1:
        
        fft_data[j_max + 1:, :] = 0
        
    if i_min<i_max:
        
        if i_min>0:
            
            fft_data[:, :i_min] = 0
            
        if i_max<k_dim-1:
            
            fft_data[:,i_max + 1:] = 0
             
    
    # Dispersion filter (wave type)
    beta = 2.28e-11
    a = 6.37e6
    spc = 24 * 3600 / (2 * np.pi * obs_per_day)
    c = np.sqrt(9.8 * np.array([h_min, h_max]))
    
    for i in range(k_dim):
        k = (i - k_dim if i > k_dim // 2 else i) / a # adjusting for circumfrence of earth
        
        freq = np.array([0,freq_dim])/spc
        j_min_wave = 0
        j_max_wave = freq_dim
        
        if wave_name.lower() == "kelvin":
            freq = k * c
        elif wave_name.lower() == "er":
            freq = -beta * k / (k**2 + 3 * beta / c)
        elif wave_name.lower() in ["mrg", "ig0"]:
            if k == 0:
                freq = np.sqrt(beta * c)
            elif k > 0:
                freq = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * beta / (k**2 * c)))
            else:
                freq = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * beta / (k**2 * c)))
        elif wave_name.lower() == "ig1":
            freq = np.sqrt(3 * beta * c + (k**2 * c**2))
        elif wave_name.lower() == "ig2":
            freq = np.sqrt(5 * beta * c + (k**2 * c**2))
        else:
            continue

        j_min_wave = int(np.floor(freq[0] * spc * time_dim)) if not np.isnan(h_min) else 0
        j_max_wave = int(np.ceil(freq[1] * spc * time_dim)) if not np.isnan(h_max) else freq_dim
        j_min_wave = min(j_min_wave, freq_dim)
        j_max_wave = max(j_max_wave, 0)
        
        # set the appropriate coefficients to zero
        
        fft_data[:j_min_wave,i] = 0
        fft_data[j_max_wave + 1:,i] = 0

    # Inverse FFT
    fft_data[:, 1:] = fft_data[:, -1:0:-1]
    
    temp_data = np.real(fft.irfft2(fft_data,axes=(1,0),s=(lon_dim,time_dim)))

    # Reconstruct full field
    filtered = in_data.copy()
    if wrap_flag:
        filtered[:, 1:] = temp_data[:, 1:]
        filtered[:, 0] = filtered[:, lon_dim-1]
    else:
        filtered[:] = temp_data
    
    if is_xarray:
        out = in_data.copy(data=temp_data)
        if "dayofyear" in out.coords:
            out = out.drop_vars("dayofyear")
        out.attrs.update({
            "wavenumber": (k_min, k_max),
            "period": (t_min, t_max),
            "depth": (h_min, h_max),
            "waveName": wave_name
        })
        return out.transpose("time", "lon")
    else:
        return temp_data
      
    
    
def extract_wave_signal(ds: xr.DataArray, wave_name='kelvin', obs_per_day=1, use_parallel=True, n_jobs=-1):
    """对OLR数据进行年循环去除并滤波提取特定波动成分"""
    
    wave_params = {
        'kelvin': {
            'freq_range': (3, 20),
            'wnum_range': (2, 14),
            'equiv_depth': (8, 90)
        },
        'er': {
            'freq_range': (9, 72),
            'wnum_range': (-10, -1),
            'equiv_depth': (8, 90)
        },
        'mrg': {
            'freq_range': (3, 10),
            'wnum_range': (-10, -1),
            'equiv_depth': (8, 90)
        },
        'ig': {
            'freq_range': (1, 14),
            'wnum_range': (1, 5),
            'equiv_depth': (8, 90)
        },
        
        'mjo': {
            'freq_range': (20, 100),
            'wnum_range': (1, 5),
            'equiv_depth': (np.nan, np.nan)
        },
        
        'td': {
            'freq_range': (1/5, 1/2.5),
            'wnum_range': (-20, -6),
            'equiv_depth': (np.nan, np.nan)
        },
    }

    assert wave_name in wave_params, f"wave_name must be one of {list(wave_params.keys())}"


    # Step 1: 年循环去除
    clim = ds.groupby('time.dayofyear').mean(dim='time')
    clim_fit = extract_low_harmonics(clim, n_harm=3)
    anomaly = ds.groupby('time.dayofyear') - clim_fit
    # check_clim(clim_fit)
    # check_ana(anomaly)
    # Step 2: 参数提取
    t_min, t_max = np.array(wave_params[wave_name]['freq_range'])
    k_min, k_max = wave_params[wave_name]['wnum_range']
    h_min, h_max = wave_params[wave_name]['equiv_depth']
    lon = clim.lon
    # Step 3: 滤波主逻辑（逐纬度调用 kf_filter）
    def _filter_lat(lat_idx):
        in_data = anomaly[:, lat_idx, :]
        return kf_filter(
            in_data.values if use_parallel else in_data,
            obs_per_day=obs_per_day,
            lon=lon,
            t_min=t_min, t_max=t_max,
            k_min=k_min, k_max=k_max,
            h_min=h_min, h_max=h_max,
            wave_name=wave_name
        )

    if use_parallel:
        filtered = Parallel(n_jobs=n_jobs)(delayed(_filter_lat)(i) for i in range(len(ds.lat)))
    else:
        filtered = [_filter_lat(i) for i in range(len(ds.lat))]

    
    filtered = np.stack(filtered, axis=1)
    # Step 4: 构造新的 DataArray
    da_filtered = xr.DataArray(
        filtered,
        coords=ds.coords,
        dims=ds.dims,
        attrs={
            'long_name': f'{wave_name.title()} Wave Component',
            'units': ds.attrs.get('units', 'unknown'),
            'wavenumber': (k_min, k_max),
            'period': (t_min, t_max),
            'depth': (h_min, h_max),
            'waveName': wave_name
        }
    )
    print(da_filtered)
    return da_filtered



ds = xr.open_dataset('I:/olr.day.mean.nc').olr.sel(
    time=slice('1980-01-01', '1999-12-31'),
    lat=slice(25, -25)).sortby('lat')

kelvin_filtered = extract_wave_signal(ds, wave_name='mjo', obs_per_day=1, use_parallel=True)

check_filter_wave(kelvin_filtered,"I:/OLRmjo_25.nc","mjo")








def check_clim(python_result):
    
    ds_ncl = xr.open_dataset("I:/cla.nc")
    clm_ncl = ds_ncl.clm
    # 对比用的纬向平均或空间平均
    clm_ncl_mean = clm_ncl.mean(dim=('lat', 'lon'))
    clim_py_mean = python_result.mean(dim=('lat', 'lon'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Python vs NCL Harmonics Climatology Comparison", fontsize=16)
    
    # 1. 平均气候态曲线对比
    ax = axes[0, 0]
    clim_py_mean.plot(ax=ax, label='Python')
    clm_ncl_mean.plot(ax=ax, label='NCL')
    ax.set_title('(a) Zonal Mean Climatology Curve')
    ax.legend()
    
    # 2. Python 版本空间图（均值沿 dayofyear）
    ax = axes[0, 1]
    python_result.mean(dim='dayofyear').plot.contourf(ax=ax, cmap='jet',levels=np.linspace(200,300,21),extend='neither')
    ax.set_title('(b) Python: Spatial Mean over DayofYear')
    
    # 3. NCL 版本空间图（均值沿 year_day）
    ax = axes[1, 0]
    clm_ncl.mean(dim='year_day').plot.contourf(ax=ax, cmap='jet',levels=np.linspace(200,300,21),extend='neither')
    ax.set_title('(c) NCL: Spatial Mean over Year_Day')
    
    # 4. 差值图（Python - NCL）
    ax = axes[1, 1]
    (python_result.mean(dim='dayofyear') - clm_ncl.mean(dim='year_day')).plot.contourf(ax=ax, cmap='RdYlBu')
    ax.set_title('(d) Difference: Python - NCL')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show() 

def check_ana(python_result):
    
    ds_ncl = xr.open_dataset("I:/ana.nc")
    clm_ncl = ds_ncl.ana
    
    random_index = np.random.randint(0, 7305)
    
    # 对比用的纬向平均或空间平均
    clm_ncl_mean = clm_ncl[:random_index].mean(dim=('lat', 'lon'))
    clim_py_mean = python_result[:random_index].mean(dim=('lat', 'lon'))
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Python vs NCL Harmonics Climatology Comparison", fontsize=16)
    # 1. 平均气候态曲线对比
    ax = axes[0, 0]
    clim_py_mean.plot(ax=ax, label='Python')
    clm_ncl_mean.plot(ax=ax, label='NCL')
    ax.set_title('(a) Zonal Mean Climatology Curve')
    ax.legend()
    
    # 2. Python 版本空间图（均值沿 dayofyear）
    ax = axes[0, 1]
    python_result[random_index].plot.contourf(ax=ax, cmap='jet',levels=np.linspace(-100, 100,21),extend='neither')
    ax.set_title('(b) Python: Climatology')
    
    # 3. NCL 版本空间图（均值沿 year_day）
    ax = axes[1, 0]
    clm_ncl[random_index].plot.contourf(ax=ax, cmap='jet',levels=np.linspace(-100, 100,21),extend='neither')
    ax.set_title('(c) NCL: Climatology')
    
    # 4. 差值图（Python - NCL）
    ax = axes[1, 1]
    (python_result[random_index] - clm_ncl[random_index]).plot.contourf(ax=ax, cmap='RdYlBu')
    ax.set_title('(d) Difference: Python - NCL')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show() 









