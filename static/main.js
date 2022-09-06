if (localStorage.getItem('darkmode') == 'yes') {
    document.body.classList.add('dark-mode')
}


document.querySelector('#btn').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    if (localStorage.getItem('darkmode') != 'yes') {
        localStorage.setItem('darkmode', 'yes')
    } else {
        localStorage.clear()
    }
})